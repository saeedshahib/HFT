from django.contrib import admin
from django.db.models import ForeignKey, OneToOneField
from django.db import models
from django.urls import reverse, re_path
from django.utils.html import format_html

from utils.widgets import CustomDateTimeWidget


# Register your models here.


class RawIdFieldForAllMixin:
    raw_id_fields = []

    def __init__(self, model, admin_site, *args, **kwargs):
        self.raw_id_fields = self.setup_raw_id_fields(model)
        super().__init__(model, admin_site, *args, **kwargs)

    def setup_raw_id_fields(self, model):
        return list(
            f.name
            for f in model._meta.get_fields()
            if isinstance(f, ForeignKey) or isinstance(f, OneToOneField)
        )


def subtract_lists(list1, list2):
    return list(set(list1) - set(list2))


def get_common_elements(list1, list2):
    differences = subtract_lists(list1, list2)
    return list(set(list1) - set(differences))


class BaseAdmin(RawIdFieldForAllMixin, admin.ModelAdmin):
    readonly_fields = ["created_at", "updated_at"]
    list_display = ('created_at', 'updated_at')
    fields = ('created_at', 'updated_at')

    model = None
    not_displayed_fields = ['trace']

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        if isinstance(db_field, models.DateTimeField):
            kwargs['widget'] = CustomDateTimeWidget()
        return super().formfield_for_dbfield(db_field, request, **kwargs)

    def get_list_display(self, request):
        if self.model is not None:
            base_fields = ['created_at', 'updated_at']
            list_unsorted = [field.name for field in self.model._meta.fields if field.name not in (base_fields + self.not_displayed_fields)]
            displayable_base_fields = [field for field in base_fields if field not in self.not_displayed_fields]
            [list_unsorted.append(field) for field in displayable_base_fields]
            return list_unsorted
        else:
            return super().get_list_display(request)

    @staticmethod
    def link_to_list_view(app_name, model):
        detail_url = reverse(f'admin:{app_name}_{model.__name__.lower()}_change', args=[0])
        return detail_url.split("/0/")[0] + "/"


class BaseAdminWithActionButtons(BaseAdmin):
    action_buttons = []

    @staticmethod
    def button_to_link(label, link, background_color="#f0b90b", text_color="black"):
        return format_html(f'<a class="button" style = "background-color: {background_color}; color:{text_color};" href="{link}">{label}</a>&nbsp;')

    readonly_fields = ['custom_actions'] + BaseAdmin.readonly_fields

    def get_list_display(self, request):
        if self.action_buttons.__len__() > 0:
            list_display = super().get_list_display(request)
            list_display.insert(1, 'custom_actions')
            return list_display
        else:
            return super().get_list_display(request)

    @staticmethod
    def append_buttons(button_array):
        button_str = ""
        for button in button_array:
            button_str += button
        return format_html(button_str)

    def custom_actions(self, obj):
        actions_str = ""
        for action_button in self.action_buttons:
            actions_str += action_button.button(obj=obj)
        return format_html(actions_str)

    def custom_actions_list(self, obj):
        actions_str = ""
        for action_button in self.action_buttons:
            actions_str += action_button.button(obj=obj)
        return format_html(actions_str)

    custom_actions.short_description = 'Actions'
    custom_actions.allow_tags = True

    custom_actions_list.short_description = 'Actions List'
    custom_actions_list.allow_tags = True

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [re_path(action.url_link(), self.admin_site.admin_view(action.action), name=action.url_name())
                       for action in self.action_buttons]
        custom_urls_list = [re_path(action.url_link(), self.admin_site.admin_view(action.action), name=action.url_name())
                            for action in self.action_buttons]
        return custom_urls + urls
